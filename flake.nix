{
  inputs = {
    # mxnet breaks with numpy >= 1.24, use numpy 1.23.5
    # https://github.com/apache/mxnet/pull/21223
    nixpkgs.url = "github:NixOS/nixpkgs/84e33aea0f7a8375c92458c5b6cad75fa1dd561b";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  outputs = {
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    devShells =
      forEachSystem
      (system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            {
              # https://devenv.sh/reference/options/
              # packages = with pkgs; [
              #   gcc
              #   zlib
              # ];

              dotenv.disableHint = true;
              languages.python = {
                enable = true;
                package = pkgs.python3.withPackages (ps:
                  with ps; [
                    # needed for showing the opencv image preview
                    (opencv4.override {enableGtk3 = true;})
                    numpy
                    torch
                    flake8
                    black
                  ]);
                venv.enable = true;
              };
            }
          ];
        };
      });
  };
}
