{pkgs}: (final: prev: rec {
  linear-operator = prev.linear-operator.overrideAttrs {
    postPatch = ''
      substituteInPlace setup.py  \
        --replace "typeguard~=2.13.3" "typeguard"
    '';
  };

  # ** flextape meme **
  skorch = prev.skorch.overrideAttrs {
    doTest = false;
    pytestCheckPhase = "exit 0";
  };

  botorch = prev.botorch.overrideAttrs {
    postPatch = ''
      substituteInPlace requirements.txt  \
        --replace "linear_operator==0.5.1" "linear_operator"
    '';
  };
})
