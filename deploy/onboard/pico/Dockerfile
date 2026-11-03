# The XRoboToolkit PC service and the PICO streamer, for a robot computer whose OS is
# older than the service's Ubuntu 22.04, e.g. the G1's Jetson (Ubuntu 20.04, glibc 2.31;
# the service and libPXREARobotSDK.so need glibc 2.35). arm64 only: the headless service
# is packaged for arm64 alone. run.sh builds and runs it; the repo is mounted at /repo, so
# a code change needs no rebuild. No GMR: the human streamer only (--target human).
FROM ubuntu:22.04

RUN [ "$(dpkg --print-architecture)" = arm64 ] \
  && apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl git g++ \
     libicu70 libglib2.0-0 libdouble-conversion3 libpcre2-16-0 libdbus-1-3 \
  && rm -rf /var/lib/apt/lists/*

# The service's files only (/opt/apps/roboticsservice): its postinst just registers desktop
# icons. Its Qt ships without Qt's own dependencies (the apt line above); the GUI, X11,
# Wayland and SQL-server plugins stay unresolved and unused.
RUN curl -fsSL -o /tmp/service.deb https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb \
  && dpkg-deb -x /tmp/service.deb / \
  && rm /tmp/service.deb

# The streamer's packages, at the repo's locked versions, and the PICO SDK at
# install_pico.sh's commit.
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv venv --python 3.11 /opt/venv \
  && uv pip install --python /opt/venv/bin/python numpy==2.4.6 pyzmq==27.2.0 msgpack==1.2.3 \
     "xrobotoolkit_sdk @ git+https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind@8941e1dbbf4313fe83846f82984eaa8c992690bf" \
  && rm -rf /root/.cache

COPY entrypoint.sh /entrypoint.sh
WORKDIR /repo
ENTRYPOINT ["/entrypoint.sh"]
