{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
              packages = with pkgs;
                [
                  gcc
                  zlib
                ]
                ++ (with python3.pkgs; [
                  (opencv4.override {
                    enableGtk3 = true;
                  })
                  torch
                  flake8
                  black
                ]);

              dotenv.disableHint = true;
              languages.python = {
                enable = true;
                venv.enable = true;
              };
            }
          ];
        };
      });
  };
}
