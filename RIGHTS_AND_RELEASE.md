# Release notes

Author and contact: Linsheng Wei (魏林笙), Communication University of China, linshengwei95@gmail.com.

Decisions made by the author (22 October 2026):

- The model weights file `weights/e3_model_state.pt` may be published together with the code.
- The weights were fine-tuned from the torchvision `mvit_v1_b` Kinetics-400 checkpoint. torchvision states that pretrained weights may have their own licences; no separate licence statement for that checkpoint was found, so this release rests on the author's decision, not on a licence document. Anyone reusing the weights should check the terms of the original checkpoint and of the Kinetics-400 data.
- The competition videos, labels, pose files and metadata files are not included and must be obtained under the competition rules.

Not included in this repository: raw videos, annotations, pose files, API tokens, SSH details, optimizer states, training code, training logs.

Licence of the code: Apache License 2.0 (`LICENSE`, `NOTICE`), chosen by the author on 22 October 2026. It applies to `infer.py`, `model.py`, `contract.py` and `compare.py`. It does not extend to the weights file, whose origin and terms are described above.
