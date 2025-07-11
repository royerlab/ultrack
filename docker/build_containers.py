import argparse
import http.client
import json
import os
from urllib.parse import urlparse


def get_ultrack_version():
    conn = http.client.HTTPSConnection("pypi.org")
    url = "/pypi/ultrack/json"
    conn.request("GET", url)
    response = conn.getresponse()

    if response.status == 200:
        data = json.loads(response.read())
        return data["info"]["version"]
    else:
        raise Exception(f"Error: Unable to fetch data (status code: {response.status})")


def fetch_tags():
    tags = []
    url = "/v2/repositories/pytorch/pytorch/tags/?page_size=100"

    while url:
        parsed_url = urlparse(url)
        host = parsed_url.netloc or "hub.docker.com"
        path = parsed_url.path + ("?" + parsed_url.query if parsed_url.query else "")

        connection = http.client.HTTPSConnection(host)
        connection.request("GET", path)
        response = connection.getresponse()

        if response.status != 200:
            print(f"Failed to fetch data: {response.status}")
            break

        data = json.loads(response.read().decode("utf-8"))
        tags.extend(data.get("results", []))
        url = data.get("next", None)

        connection.close()

    return tags


def parse_tags(tags):
    cuda_versions = {}

    for tag in tags:
        name = tag["name"]
        if "-" in name:
            parts = name.split("-")
            version = parts[0]
            if len(parts) > 1 and "cuda" in parts[1]:
                cuda_version = parts[1].split("cuda")[1].split("-")[0]
                cudnn_version = (
                    parts[2].split("cudnn")[1]
                    if len(parts) > 2 and "cudnn" in parts[2]
                    else "unknown"
                )
                if version not in cuda_versions:
                    cuda_versions[version] = []
                if (cuda_version, cudnn_version) not in cuda_versions[version]:
                    cuda_versions[version].append((cuda_version, cudnn_version))

    return cuda_versions


def available_versions():
    tags = fetch_tags()

    if not tags:
        raise Exception("No tags found.")

    cuda_versions = parse_tags(tags)
    if not cuda_versions:
        raise Exception("No CUDA versions found.")

    # Find the latest version
    latest_version = max(
        cuda_versions.keys(), key=lambda v: list(map(int, v.split(".")))
    )

    return [
        (latest_version, cuda, cudnn) for cuda, cudnn in cuda_versions[latest_version]
    ]


def build_image(image_type, ultrack_version, docker_torch_tag=None, cuda=None):
    if image_type == "cpu":
        os.system(
            f"docker build -t royerlab/ultrack:{ultrack_version}-cpu "
            f"--build-arg ULTRACK_VERSION={ultrack_version} cpu"
        )
    elif image_type == "gpu" and docker_torch_tag and cuda:
        os.system(
            f"docker build -t royerlab/ultrack:{ultrack_version}-cuda{cuda} "
            f"--build-arg PYTORCH_VERSION={docker_torch_tag} "
            f"--build-arg ULTRACK_VERSION={ultrack_version} gpu"
        )


def main():
    parser = argparse.ArgumentParser(description="Docker image builder for Ultrack.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Image type to build (e.g., cpu, cuda<version>) or '--all' for all images.",
    )

    args = parser.parse_args()
    ultrack_version = get_ultrack_version()

    if not args.image:
        print(
            f"Error: No argument provided. Available options to build ultrack version {ultrack_version}:"
        )
        print("all: Build all images")
        print("cpu: Build CPU image")
        for latest_torch, cuda, cudnn in available_versions():
            print(
                f"cuda{cuda}: Build GPU image for PyTorch {latest_torch} with CUDA {cuda} and cuDNN {cudnn}"
            )
        return

    if args.image == "all":
        build_image("cpu", ultrack_version)
        for latest_torch, cuda, cudnn in available_versions():
            docker_torch_tag = f"{latest_torch}-cuda{cuda}-cudnn{cudnn}"
            build_image("gpu", ultrack_version, docker_torch_tag, cuda)
    elif args.image == "cpu":
        build_image("cpu", ultrack_version)
    elif args.image.startswith("cuda"):
        cuda_version = args.image[4:]
        for latest_torch, cuda, cudnn in available_versions():
            if cuda == cuda_version:
                docker_torch_tag = f"{latest_torch}-cuda{cuda}-cudnn{cudnn}"
                build_image("gpu", ultrack_version, docker_torch_tag, cuda)
                break
        else:
            print(
                f"Error: CUDA version {cuda_version} not found in available versions."
            )
    else:
        print(
            f"Error: Unknown argument '{args.image}'. Use 'all', 'cpu', or 'cuda<version>'."
        )


if __name__ == "__main__":
    main()
