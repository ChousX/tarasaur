{
        description = "A flake using Oxalica's rust-overlay for Bevy development.";

        inputs = {
                nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
                rust-overlay = {
                        url = "github:oxalica/rust-overlay";
                        inputs.nixpkgs.follows = "nixpkgs";
                };
        };

        outputs =
                {
                        nixpkgs,
                        rust-overlay,
                        ...
                }:
                let
                        supportedSystems = [
                                "x86_64-linux"
                                "aarch64-linux"
                        ];
                        forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
                in
                {
                        devShells = forAllSystems (
                                system:
                                let
                                        pkgs = import nixpkgs {
                                                inherit system;
                                                overlays = [ (import rust-overlay) ];
                                        };

                                        rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
                                                extensions = [
                                                        "rust-src"
                                                        "rust-analyzer"
                                                ];
                                        };

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
                                        default = pkgs.mkShell {
                                                name = "bevy";
                                                nativeBuildInputs = [ pkgs.pkg-config ];
                                                buildInputs = buildInputs;
                                                packages = [
                                                        rustToolchain
                                                        pkgs.bacon
                                                        pkgs.renderdoc
                                                ];
                                                LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
                                        };
                                }
                        );
                };
}
