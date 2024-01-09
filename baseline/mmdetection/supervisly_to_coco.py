import ast
import json
import argparse


def args_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a list")
    return v


def canAppend(range_tuple, cur_idx, cur_img_id):
    for idx, (x, y) in enumerate(range_tuple):
        if idx != cur_idx:
            if x <= cur_img_id <= y:
                return False
    return True


def convert(args):
    dir_paths = []
    for output in args.outputs:
        dir_paths.append(args.root + output)

    with open(dir_paths[0], "r") as file:
        json_data = json.load(file)

    new_annotations = []
    annotation_id = 0
    tmpSet = set()
    for idx, dir_path in enumerate(dir_paths):
        with open(dir_path, "r") as file:
            json_data = json.load(file)
            # make image id start from 0
            start_image_id = json_data["images"][0]["id"]
            for i in range(len(json_data["images"])):
                json_data["images"][i]["id"] -= start_image_id

            # make category id start from 0
            for i in range(len(json_data["categories"])):
                json_data["categories"][i]["id"] -= 1

            for i in range(len(json_data["annotations"])):
                json_data["annotations"][i]["image_id"] -= start_image_id
                json_data["annotations"][i]["category_id"] -= 1

            if idx == 0:
                new_json_data = json_data

            for ann_num in range(len(json_data["annotations"])):
                if canAppend(
                    args.range, idx, json_data["annotations"][ann_num]["image_id"]
                ):
                    if json_data["annotations"][ann_num]["image_id"] not in tmpSet:
                        tmpSet.add(json_data["annotations"][ann_num]["image_id"])
                    json_data["annotations"][ann_num]["id"] = annotation_id
                    annotation_id += 1
                    new_annotations.append(json_data["annotations"][ann_num])
        file.close()

    new_json_data["annotations"] = new_annotations
    out_path = args.root + args.output

    with open(out_path, "w") as file:
        json.dump(new_json_data, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # root
    parser.add_argument(
        "--root",
        type=str,
        default="../../dataset/",
        help="dataset's location (default: ../../dataset/)",
    )

    # # of outputs
    parser.add_argument(
        "--number_of_outputs",
        type=int,
        default=2,
        help="enter number of supervisly output",
    )

    # outputs
    parser.add_argument(
        "--outputs", type=args_as_list, default=[], help="Enter list of outputs"
    )

    # range of output
    parser.add_argument(
        "--range",
        type=args_as_list,
        default=[],
        help="Enter range of each output,\n\
            ex) output1's range(0~1000), output2's range(1001~4883), input is [0,1000,1001,4883]",
    )

    # nameof output
    parser.add_argument(
        "--output", type=str, default="clean.json", help="name of output json file"
    )
    args = parser.parse_args()

    outputs_range = []

    for i in range(1, len(args.range), 2):
        outputs_range.append((args.range[i - 1], args.range[i]))

    args.range = outputs_range
    if args.outputs == []:
        raise Exception("Outputs are empty")

    if args.number_of_outputs != len(args.outputs):
        print(f"{args.number_of_outputs}!={len(args.outputs)}")
        raise Exception("# of outputs and len(outputs) mismatch")

    if len(args.outputs) != len(args.range):
        raise Exception("len(outputs) and len(range) mismatch")

    convert(args)
