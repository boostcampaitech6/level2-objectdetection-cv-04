import ast
import json
import argparse


def args_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a list")
    return v


def is_available(range_tuple, cur_idx, cur_img_id):
    for idx, (x, y) in enumerate(range_tuple):
        if idx != cur_idx:
            if x <= cur_img_id <= y:
                return False
    return True


def convert(args):
    dir_paths = []
    for input in args.inputs:
        dir_paths.append(args.root + input)

    with open(dir_paths[0], "r") as file:
        json_data = json.load(file)

    new_annotations = []
    annotation_id = 0

    for idx, dir_path in enumerate(dir_paths):
        with open(dir_path, "r") as file:
            json_data = json.load(file)
            # make image id start from 0
            start_image_id = json_data["images"][0]["id"]
            for i in range(len(json_data["images"])):
                json_data["images"][i]["id"] -= start_image_id
                json_data["images"][i]["file_name"] = "train/"+json_data["images"][i]["file_name"]

            # make category id start from 0
            for i in range(len(json_data["categories"])):
                json_data["categories"][i]["id"] -= 1

            for i in range(len(json_data["annotations"])):
                json_data["annotations"][i]["image_id"] -= start_image_id
                json_data["annotations"][i]["category_id"] -= 1

            if idx == 0:
                new_json_data = json_data

            for ann_num in range(len(json_data["annotations"])):
                if is_available(
                    args.range, idx, json_data["annotations"][ann_num]["image_id"]
                ):
                    json_data["annotations"][ann_num]["id"] = annotation_id
                    annotation_id += 1
                    new_annotations.append(json_data["annotations"][ann_num])
        file.close()

    new_json_data["annotations"] = new_annotations
    out_path = args.root + args.output

    with open(out_path, "w") as file:
        json.dump(new_json_data, file, indent=2)
    print(f"done, {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # root
    parser.add_argument(
        "--root",
        type=str,
        default="../../dataset/",
        help="dataset's location (default: ../../dataset/)",
    )

    # inputs
    parser.add_argument(
        "--inputs",
        type=args_as_list,
        default=[],
        help="Enter list of inputs ex) ['instances.json','instances2.json','instances.json']",
    )

    # range of input
    parser.add_argument(
        "--range",
        type=args_as_list,
        default=[],
        help="Enter range of each input,\n\
            ex) input1's range(0~1000), input2's range(1001~4883), \n\
                input is [0,1600,1601,2400,2401,4883]",
    )

    # name of output
    parser.add_argument(
        "--output", type=str, default="clean.json", help="name of output json file"
    )

    args = parser.parse_args()

    inputs_range = []

    for i in range(1, len(args.range), 2):
        inputs_range.append((args.range[i - 1], args.range[i]))

    args.range = inputs_range

    if args.inputs == []:
        raise Exception("Inputs are empty")

    if len(args.inputs) != len(args.range):
        raise Exception("len(outputs) and len(range) mismatch")

    convert(args)
