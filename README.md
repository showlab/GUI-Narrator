## GUI Action Narrator: Where and When Did That Action Take Place?

Qinchen Wu, Difei Gao, Kevin Qinghong Lin, Zhuoyu Wu, Xiangwu Guo, Peiran Li, Weichen Zhang, Hengxu Wang, Mike Zheng Shou

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://showlab.github.io/GUI-Narrator/)

## 🤖: Introduction

We introduce GUI action dataset **Act2Cap** as well as an effective framework: **GUI Narrator** for GUI video captioning that utilizes the cursor as a visual prompt to enhance the interpretation of high-resolution screenshots.

## 📑: Events

- 19 Jun 2024: We release our paper on Arxiv.
- 15 Aug 2024: The automatic collected datasets and human demonstration datasets are available.

---

- Download **ACT2CAP** dataset, which consists of 10-frame GUI screenshot sequences depicting atomic actions. **[**Download link here**](**https://drive.google.com/file/d/18cL3ByBkEMI-eTKrelaEXWeiF3QwZAAl/view?usp=drive_link**)**.
- Narrations based on 10 frames screenshots in `.data_annotation` . Please replace the  `<path>`  placeholder with the root path of ACT2CAP image files
    ```{
    "id": "identity_3",
    "conversations": [
        {
        "from": "user",
        "value": "Picture1: <img>.<path>/action_video_10_frames/x/a_prompt.png</img>\n
        Picture2: <img>.<path>/action_video_10_frames/x/b_prompt.png</img>\n
        Picture3: <img>.<path>/action_video_10_frames/x/a_crop.png</img>\n
        Picture4: <img>.<path>/action_video_10_frames/x/b_crop.png</img>\n 
        the images shows video clips of an atomic action on graphic user interface. The cursor is acting in the green bounding box\nDescribe what is the cursor doing based on the given images. Leftclick, Rightclick, Doubleclick, Type write or Drag."
        },
        {
        "from": "assistant",
        "value": "The cursor LeftClick on Swap"
        }
    ]
    }
    ```
    Where `a`, `b` denotes the start and the end frame index respectively. `x` denotes the folder index.
    The terms `Prompt` and `Crop` refers to screen shot with visual prompt and cropped detailed images generated depend on cursor detection module. 
    However, if you are interested in the original images, you can substitute them with `frame_idx`. 
