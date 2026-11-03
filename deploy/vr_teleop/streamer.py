"""The PICO streamer process: headset samples -> button state machine -> calibrated
frames at 50 Hz -> ZMQ PUB chunks for the sim2sim policy node (`deploy/sim2sim/teleop.py`).
The target makes the frames: GMR's retargeting to the robot's reference states
(`--target gmr`, the default: `gmr_streamer.GmrTarget`, for the motion interface) or the
operator on the uniform human skeleton (`--target human`: `human_target.HumanTarget`, for
the human interface; GMR is never loaded).

Each `Streamer.step` pulls the newest sample from a source (`PicoSource` or
`ReplaySource`), advances the `Controls` state machine, calibrates on the start combo,
resamples the body onto the 50 Hz device-time grid and retargets every new frame. The
last `CHUNK` frames are kept in a ring and republished with every message, so a dropped
packet costs nothing: the sim de-duplicates by `frame_index`, which counts up from 0 at
process start and never resets (not even on recalibration).

Run: `python -m deploy.vr_teleop.streamer --robot g1|x2|t2 [--replay F] [--record F]
[--port N]`, or `--target human` (no robot).
"""

from __future__ import annotations

import argparse
import collections
import signal
import threading
import time
from collections.abc import Callable

from deploy.vr_teleop.constants import CHUNK, PORT
from deploy.vr_teleop.controls import Controls
from deploy.vr_teleop.pico_source import PicoSource, Recorder, ReplaySource
from deploy.vr_teleop.resample import Resampler

MAX_GAP_S = 0.2  # Stream gap (and backward clock jump) that restarts the resampler.
_NS = 1_000_000_000


class Streamer:
  """`target` turns PICO bodies into frames and messages (`gmr_streamer.GmrTarget`,
  `human_target.HumanTarget`):
  `calibrated`, `calibrate(body)`, `frame(index, body)`, `message(mode, calib_id, t_ns,
  fps, frames) -> (message, bytes)`."""

  def __init__(self, source, publish: Callable[[bytes], None], target) -> None:
    self.source = source
    self.publish = publish
    self.controls = Controls()
    self.target = target
    self._resampler = Resampler(max_gap_s=MAX_GAP_S)
    self._ring: collections.deque = collections.deque(maxlen=CHUNK)
    self._next_index = 0
    self._seen_t_ns: int | None = None  # Last polled sample, to skip repeats.
    self._last_t_ns: int | None = None  # Newest device time the resampler has taken.
    self._frame_t_ns: int | None = (
      None  # Device time of the sample behind the newest frame.
    )

  def step(self, now_ns: int):
    """Process the newest sample, if it is new, and publish; `now_ns` is wall time.

    `Controls` runs on wall time, not device time, so its combo-hold timer survives a
    headset clock reset. Returns the published message, or None if there was no new
    sample (the sim's timeout then sees silence, not a repeated stale message).
    """
    sample = self.source.latest()
    if sample is None or sample.t_ns == self._seen_t_ns:
      return None
    self._seen_t_ns = sample.t_ns

    event = self.controls.update(sample.buttons, now_ns / _NS)
    if event == "calibrate":
      self.target.calibrate(sample.body)
      self._ring.clear()  # Frames from the previous calibration must not be resent.

    # Clock reset (headset or app restart): a clock far behind the last sample would be
    # dropped as stale until it caught up, so start a fresh grid instead.
    if self._last_t_ns is not None and sample.t_ns < self._last_t_ns - MAX_GAP_S * _NS:
      self._resampler = Resampler(max_gap_s=MAX_GAP_S)
      self._last_t_ns = None
    if self._last_t_ns is None or sample.t_ns > self._last_t_ns:
      self._last_t_ns = sample.t_ns

    for body in self._resampler.push(sample.t_ns, sample.body):
      if not self.target.calibrated:
        continue
      self._ring.append(self.target.frame(self._next_index, body))
      self._next_index += 1
      self._frame_t_ns = sample.t_ns

    msg, data = self.target.message(
      self.controls.mode, self.controls.calib_id,
      self._frame_t_ns if self._ring else sample.t_ns, float(self.source.fps),
      list(self._ring),
    )
    self.publish(data)
    return msg


def main() -> None:
  import zmq  # Lazy: only the process that publishes needs it.

  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument(
    "--replay", metavar="F", help="play a recording instead of the headset"
  )
  p.add_argument("--record", metavar="F", help="save the incoming samples to F (.npz)")
  p.add_argument("--port", type=int, default=PORT)
  p.add_argument("--target", choices=["gmr", "human"], default="gmr",
                 help="gmr: the robot's reference states (motion interface); human: the"
                 " operator on the uniform skeleton (human interface, no GMR)")
  p.add_argument("--robot", choices=["g1", "x2", "t2"],
                 help="--target gmr: the robot GMR retargets to (the policy node's)")
  p.add_argument("--human-height", type=float, default=None,
                 help="--target gmr: operator height (m) for GMR's scaling (default: its"
                 " 1.8 m; the human target measures the operator at calibration)")
  p.add_argument(
    "--rate-hz", type=float, default=100.0, help="polling rate of the loop"
  )
  args = p.parse_args()
  if args.target == "gmr" and args.robot is None:
    p.error("--target gmr needs --robot")

  source = ReplaySource(args.replay, realtime=True) if args.replay else PicoSource()
  recorder = Recorder(args.record) if args.record else None
  if recorder is not None:
    source.on_sample = recorder.add

  sock = zmq.Context.instance().socket(zmq.PUB)
  sock.bind(f"tcp://*:{args.port}")
  if args.target == "human":
    from deploy.vr_teleop.human_target import HumanTarget

    target = HumanTarget()
  else:
    from deploy.vr_teleop import gmr_streamer  # Lazy: GMR takes a few seconds to load.

    target = gmr_streamer.GmrTarget(args.robot, args.human_height)
  streamer = Streamer(source, sock.send, target)
  source.start()
  # Ctrl-C and SIGTERM only set a flag, so a repeated signal cannot interrupt the
  # cleanup that saves the recording.
  stop = threading.Event()
  for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: stop.set())
  print(
    f"[streamer] publishing on tcp://*:{args.port} at <= {args.rate_hz:g} Hz",
    flush=True,
  )

  period = 1.0 / args.rate_hz
  next_tick = last_report = time.monotonic()
  announced_end = False
  try:
    while not stop.is_set():
      streamer.step(time.monotonic_ns())
      now = time.monotonic()
      if now - last_report >= 5.0:
        last_report = now
        print(
          f"[streamer] pico {source.fps:.1f} Hz, mode {streamer.controls.mode}, "
          f"calib_id {streamer.controls.calib_id}",
          flush=True,
        )
      if getattr(source, "finished", False) and not announced_end:
        announced_end = True
        print("[streamer] replay finished; holding the last sample", flush=True)
      next_tick += period
      stop.wait(max(0.0, next_tick - time.monotonic()))
  finally:
    # Save the recording however the loop ended (Ctrl-C, SIGTERM or an exception),
    # even if stopping the source fails; the loop's own exception still propagates.
    try:
      source.stop()
    except Exception as e:  # noqa: BLE001 - reported; the recording comes first.
      print(f"[streamer] stopping the source failed: {e!r}", flush=True)
    if recorder is not None:
      recorder.close()
      print(f"[streamer] saved {len(recorder)} samples to {recorder.path}", flush=True)
    sock.close()


if __name__ == "__main__":
  main()
