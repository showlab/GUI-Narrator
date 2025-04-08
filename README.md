## GUI Action Narrator: Where and When Did That Action Take Place?

Qinchen Wu, Difei Gao, Kevin Qinghong Lin, Zhuoyu Wu, Xiangwu Guo, Peiran Li, Weichen Zhang, Hengxu Wang, Mike Zheng Shou

<!-- [![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://showlab.github.io/GUI-Narrator/) -->

## 🤖: Introduction

We introduce GUI action dataset **Act2Cap** as well as an effective framework: **GUI Narrator** for GUI video captioning that utilizes the cursor detection to enhance the interpretation of high-resolution screenshots and keyframe extraction in GUI actions.


## 📋 ToDo List

- [x] Model for Cursor detector and Narrator
- [ ] Code of conduct


-- Our model and test benchmark are availble on  [![Hugging Face](https://img.shields.io/badge/Demo-HuggingFace-blue)](https://huggingface.co/FRank62Wu/ShowUI-Narrator).






<!-- - Download **ACT2CAP** dataset, which consists of 10-frame GUI screenshot sequences depicting atomic actions. **[Download link here](https://drive.google.com/file/d/18cL3ByBkEMI-eTKrelaEXWeiF3QwZAAl/view?usp=drive_link)**.
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
---
- Download **Cursor detection and Key frame extraction checkpoint** from **[Download link here](https://drive.google.com/file/d/1ChrpBuPL7W84mKNsSsbueff5EGlyB3h2/view?usp=sharing)**

- Import supporting packages
  ```
  pip install -r requirements.txt
  ```

- Run inference code as below, the visual prompts and cropped images will be generated in folder `frames_sample `
   ``` 
       cd model
       python run_model.py \
       --frame_extract_model_path /path/to/checkpoint_key_frames \
       --yolo_model_path /path/to/Yolo_best \
       --images_path /path/to/frames_sample 
   ``` -->