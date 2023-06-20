from duckduckgo_search import ddg_images
from fastai.vision.all import download_images, resize_images, verify_images, get_image_files, ImageBlock, \
    CategoryBlock, RandomSplitter, parent_label, ResizeMethod, Resize, vision_learner, resnet18, error_rate, \
    L, Path, DataBlock


def search_images(search_term, max_images=30):
    print(f"Searching for '{search_term}'")
    return L(ddg_images(search_term, max_results=max_images)).itemgot('image')


def search_and_populate(search_term, category, file_path, max_images=30):
    dest = (file_path/category)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{search_term} photo', max_images=max_images))
    resize_images(file_path/category, max_size=400, dest=file_path/category)


path = Path('seefood')
search_and_populate("hotdog", "hotdog", path, max_images=90)
for o in ['burger', 'sandwich', 'fruit', 'chips', 'salad']:
    search_and_populate(o, "not_hotdog", path, max_images=30)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"{len(failed)} failed images")

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=parent_label,
    item_tfms=[Resize(256, ResizeMethod.Squish)]
).dataloaders(path, bs=32)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

learn.export("hotdogModel.pkl")
