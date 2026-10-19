{ pkgs, lib, ... }:
let
	buildInputs = with pkgs; [
		udev
		alsa-lib
		wayland
		libxkbcommon
	];

	runtimeLibs =
		with pkgs;
		[
			vulkan-loader
			libX11
			libXcursor
			libXrandr
			libXi
		]
		++ buildInputs;
in
{
	# Pulled in by languages.rust; stdenv already provides the C toolchain.
	languages.c.enable = false;

	languages.rust = {
		enable = true;
		channel = "nightly";
		components = [
			"rustc"
			"cargo"
			"clippy"
			"rustfmt"
			"rust-analyzer"
			"rust-src"
		];
	};

	packages =
		with pkgs;
		[
			pkg-config
			bacon
			renderdoc
			# Must match the Tracy protocol Bevy's `trace_tracy` uses.
			(callPackage ./nix/tracy.nix { })
		]
		++ buildInputs;

	env.LD_LIBRARY_PATH = lib.makeLibraryPath runtimeLibs;
}
