#!/bin/sh
# The bench on SteamOS, installed as ~/tarasaur-bench/bench by
# tools/bench-deck.sh. Run it there, or through that script from the build host.
#
# The binary is built on another host against a newer glibc than SteamOS's, so
# it runs under the loader and C runtime bundled in lib/ and takes everything
# else from SteamOS (/usr/lib): Vulkan, the GPU driver, Wayland and udev.
# --library-path covers only the bench, so anything it shells out to still gets
# the system libraries.
#
# It renders inside a nested gamescope, which is how a game runs on this
# machine: a real swapchain, present and composite path, so a frame time here
# is one you could ship. `-w/-h` fix the render resolution whatever the host
# does with gamescope's own window, which is what `pool-ab.md`'s "the window
# came up 1080x800" hazard needed.
#
# `-r 240` is load-bearing. It advertises a 240 Hz nested output, so nothing
# paces the bench near 60: measured with vkmark on this Deck, the same scene
# ran at 181 fps nested without it and 719 fps with (2645 fps headless, which
# skips the composite a real game pays for). That matters at a 60 fps target,
# where a silent 16.6 ms vsync cap would look exactly like success.
#
# Without `--expose-wayland` gamescope unsets WAYLAND_DISPLAY for its child
# and hands it `DISPLAY=:1`, its own Xwayland, so winit takes the X11 path
# into gamescope rather than connecting past it to the host compositor.
#
#   GAMESCOPE  the wrapper, as one word-split string. Empty runs the bench
#              bare. To prove a cap is the host compositor's:
#              GAMESCOPE="gamescope --backend headless -W 1280 -H 800 -w 1280 -h 800 --"
#              To put it on the Deck's own panel at its real refresh:
#              GAMESCOPE="gamescope --backend drm -W 1280 -H 800 --"
#   ALLOW_BATTERY=1  bench on battery anyway (slower, and it throttles)
set -u
d=$(dirname "$(readlink -f "$0")")

# A Deck on battery clocks down and holds a lower TDP, so the numbers aren't
# comparable with a plugged-in run.
if [ -z "${ALLOW_BATTERY:-}" ] && [ -r /sys/class/power_supply/ACAD/online ]; then
	if [ "$(cat /sys/class/power_supply/ACAD/online)" != 1 ]; then
		echo "on battery: plug in, or pass ALLOW_BATTERY=1" >&2
		exit 1
	fi
fi

# Keep the Deck awake while it measures: logind's idle lock, and in Desktop
# mode Plasma's power management too (an ssh session holds neither on its own).
#
# `--what=idle` only: taking the sleep lock as well needs interactive polkit
# auth here, which an ssh session can't give, and the whole run would die
# before it started.
if [ -z "${TARASAUR_AWAKE:-}" ]; then
	export TARASAUR_AWAKE=1
	inhibit="systemd-inhibit --what=idle --who=tarasaur --why=bench"
	$inhibit true 2>/dev/null || {
		echo "note: can't take an idle lock; the Deck may blank mid-run" >&2
		inhibit=""
	}
	bus=/run/user/$(id -u)/bus
	if [ -S "$bus" ] && command -v kde-inhibit >/dev/null && pgrep -x plasmashell >/dev/null; then
		# shellcheck disable=SC2086
		exec $inhibit env DBUS_SESSION_BUS_ADDRESS="unix:path=$bus" \
			kde-inhibit --power --screenSaver "$0" "$@"
	fi
	# shellcheck disable=SC2086
	exec $inhibit "$0" "$@"
fi

# gamescope needs these to reach the session's wayland socket, which an ssh
# session doesn't have in its environment.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
# 1280x800 is the bench's own window size: the Deck's native panel.
wrap="gamescope --backend wayland -W 1280 -H 800 -w 1280 -h 800 -r 240 -b --force-windows-fullscreen --rt --"
[ "${GAMESCOPE+set}" = set ] && wrap=$GAMESCOPE

# The bench's own default scratch world is under the checkout it was built
# from, which isn't here, so supply one unless the caller did.
case " $* " in
	*" --world-dir "*) ;;
	*) set -- "$@" --world-dir "$HOME/.cache/tarasaur-bench-world" ;;
esac

# sysmon.sh matches DRM fdinfo clients against one pid, so that pid has to be
# the bench itself. Every wrapper here is `exec`ed, but gamescope *forks* its
# child, so the pid run.sh would see is gamescope's and the bench's GPU time
# would land in "other". When run.sh asks us to (BENCH_PIDFILE), the innermost
# shell records its own pid and then becomes the bench.
# shellcheck disable=SC2086 # $wrap is a command line, split on purpose
if [ -n "${BENCH_PIDFILE:-}" ]; then
	exec $wrap sh -c 'echo $$ >"$1"; shift; exec "$@"' sh "$BENCH_PIDFILE" \
		"$d/lib/ld-linux-x86-64.so.2" --library-path "$d/lib:/usr/lib" \
		"$d/bin/bench" "$@"
fi
exec $wrap "$d/lib/ld-linux-x86-64.so.2" --library-path "$d/lib:/usr/lib" \
	"$d/bin/bench" "$@"
