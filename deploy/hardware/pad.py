"""Button sources for the hardware nodes: a Linux joystick pad and a keyboard fallback.

Both give `held() -> set[str]` over the five names the nodes' state machines use (the
physical-remote names of `g1_node.py`): `start`, `A`, `Y`, `B`, `select`. A node polls
`held()` once per 50 Hz tick and does its own edge detection.

`Pad` reads a joystick device (`/dev/input/jsN`, the Linux joystick API: 8-byte
`js_event` records, `time u32, value s16, type u8, number u8`) without blocking. The
button numbers are the xpad (Xbox) and DualShock (PlayStation) ones in `LAYOUTS`. The
device's initial-state replay (the `0x80` init flag) is ignored, so a button already
held when the node starts counts once it is released and pressed again. If the device
disappears (the pad's battery dies, the dongle is unplugged) the pad reports nothing
held from then on, which the nodes treat as no input, not as a press. `dump=True`
prints every raw button event as `[pad] button N down|up`, to find the numbers of an
unknown pad.

`Keys` is the fallback when there is no pad, over the terminal the node runs in (cbreak
mode, so nothing is echoed and Ctrl-C still works): Enter is `start`, `a` `A`, `y` `Y`,
`b` `B`, `q` `select`. A key counts once per press, so a node sees a one-tick pulse
where a pad holds the button; it is disabled (`held()` always empty) when the input is
not a terminal (over ssh: `ssh -t`). `flush()` drops the keys typed so far, so a key
pressed while a node starts up is not read as a press at its first tick.
"""

from __future__ import annotations

import errno
import os
import select
import struct
import termios
import tty

# Button numbers under the Linux joystick API (xpad and DualShock drivers).
LAYOUTS: dict[str, dict[str, int]] = {
  "xbox": {"A": 0, "B": 1, "Y": 3, "select": 6, "start": 7},
  "ps": {"A": 0, "B": 1, "Y": 3, "select": 8, "start": 9},
}

BUTTON = 0x01  # js_event type; the init flag 0x80 is ORed on the initial-state replay
GONE = {errno.ENODEV, errno.EIO, errno.EBADF}  # the device vanished under the read

KEYS = {"\r": "start", "\n": "start", "a": "A", "y": "Y", "b": "B", "q": "select"}


class Pad:
  """A joystick device (or any fd of `js_event` records), polled without blocking."""

  def __init__(self, fd: int, layout: str = "xbox", dump: bool = False, name: str = "pad"):
    self.fd = fd
    os.set_blocking(fd, False)  # `held` drains until the read would block, on any fd
    self.buttons = LAYOUTS[layout]
    self.dump = dump
    self.name = name
    self.alive = True
    self.down: set[int] = set()

  @classmethod
  def open(cls, device: str = "/dev/input/js0", layout: str = "xbox", dump: bool = False) -> Pad:
    return cls(os.open(device, os.O_RDONLY | os.O_NONBLOCK), layout, dump, name=device)

  def held(self) -> set[str]:
    """The names of the buttons down after every pending event."""
    while self.alive:
      try:
        data = os.read(self.fd, 8 * 64)
      except BlockingIOError:
        break
      except OSError as e:
        if e.errno not in GONE:
          raise
        data = b""
      if not data:  # end of file: the device is gone
        self.alive = False
        self.down.clear()
        print(f"[pad] {self.name} gone; no buttons")
        break
      for _, value, kind, number in struct.iter_unpack("IhBB", data):
        if kind != BUTTON:
          continue
        if self.dump:
          print(f"[pad] button {number} {'down' if value else 'up'}")
        if value:
          self.down.add(number)
        else:
          self.down.discard(number)
    return {name for name, number in self.buttons.items() if number in self.down}

  def close(self) -> None:
    os.close(self.fd)


class Keys:
  """Terminal keys as buttons: each press shows in one `held()` call."""

  def __init__(self, fd: int | None = None):
    self.fd = 0 if fd is None else fd  # stdin; `isatty` is False on a closed or replaced one
    self.enabled = os.isatty(self.fd)
    if self.enabled:
      self.saved = termios.tcgetattr(self.fd)
      tty.setcbreak(self.fd)

  def held(self) -> set[str]:
    """The names of the keys pressed since the last call."""
    if not self.enabled or not select.select([self.fd], [], [], 0)[0]:
      return set()
    return {KEYS[key] for key in os.read(self.fd, 64).decode(errors="ignore") if key in KEYS}

  def flush(self) -> None:
    """Drop the keys typed and not yet read."""
    if self.enabled:
      termios.tcflush(self.fd, termios.TCIFLUSH)

  def close(self) -> None:
    # Flush, so keys typed after the last tick do not land in the shell.
    if self.enabled:
      termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.saved)
