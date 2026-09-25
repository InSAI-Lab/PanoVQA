import re

# Define placeholders for dataset paths
# Example: 
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "../PanoVQA/PanoVQA_mini/NuScenes",
}

# PanoVQA datasets:
NuScenes_mini_train = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes/train.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

NuScenes_mini_val = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes/val.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

DeepAccident_mini_train = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/DeepAccident/train.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

DeepAccident_mini_val = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/DeepAccident/val.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

DeepAccident_train = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA/DeepAccident/train.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}
DeepAccident_val = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA/DeepAccident/val.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

BlendPASS_val = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/BlendPASS/val.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

NuScenes_train = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA/NuScenes/train.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}
NuScenes_val = {
    "annotation_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA/NuScenes/val.json",
    "data_path": "/home/hk-project-pai00053/wd1434/programmes/PLM/PanoVQA/PanoVQA_mini/NuScenes"
}

data_dict = {
    "NuScenes_mini_train": NuScenes_mini_train,
    "DeepAccident_mini_train": DeepAccident_mini_train,

    # val:
    "NuScenes_mini_val": NuScenes_mini_val,
    "DeepAccident_mini_val": DeepAccident_mini_val,
    "OASS_val": BlendPASS_val,

    # full datasets:
    "NuScenes_train": NuScenes_train,
    "DeepAccident_train": DeepAccident_train,
}

val_pano_whole_data_dict = {
    "NuScenes_val": NuScenes_val,
    "DeepAccident_val": DeepAccident_val,
    "OASS_val": BlendPASS_val,
}

val_pano_mini_data_dict = {
    "NuScenes_mini_val": NuScenes_mini_val,
    "DeepAccident_mini_val": DeepAccident_mini_val,
    "OASS_val": BlendPASS_val,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    dataset_use = "pana_test%80".split(",")
    dataset_names = dataset_use

    print("dataset_names:", dataset_names)
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
