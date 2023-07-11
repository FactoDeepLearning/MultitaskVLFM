from setuptools import setup, find_namespace_packages

setup(name='MultitaskVLFM',
      packages=find_namespace_packages(include=["Multitasking", "Multitasking.*"]),
      version='1.0.0',
      install_requires=[
            "torch==1.13.0",
            "torchvision==0.14.0",
            "tensorboard",
            "scikit-learn",
            "clip",
            "open_clip_torch",
            "transformers",
            "tqdm",
            "pillow",
            "einops",
            "wget",
            "clip @ git+https://github.com/openai/CLIP.git"
            ]
      )