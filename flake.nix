{
  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      #Didn't test this to spare my poor CPU
      #config.cudaSupport = true;
    };
  in {
    devShells.${system}.default = let
      pythonOverrides = pkgs.callPackage ./requirements.nix {inherit pkgs;};
      python = pkgs.python311.override {packageOverrides = pythonOverrides;};
      pythonEnv = python.withPackages (ps:
        with ps; [
          xgboost
          pyarrow
          packaging
          pip
          numpy
          ipykernel
          jupyter-core
          ipywidgets
          scikit-learn
          notebook
          torch
          torchinfo
          botorch
          skorch
          ax
          opencv4
          matplotlib
          joblib
        ]);
    in
      pkgs.mkShell {
        packages = [pythonEnv];
      };
  };
}
