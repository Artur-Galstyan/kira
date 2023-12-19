import tempfile
import time

import bs4
import requests
from google.cloud import storage
from tqdm import tqdm


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    generation_match_precondition = 0

    try:
        blob.upload_from_filename(
            source_file_name, if_generation_match=generation_match_precondition
        )
    except Exception as e:
        pass
        # probably already exists, ignore

    # print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def main():
    bucket_name = "machine-learning-training-datasets"
    root = "https://www.vgmusic.com/"
    page = requests.get(root)

    soup = bs4.BeautifulSoup(page.content, "html.parser")

    links = soup.find_all("a")

    links = list(
        filter(
            lambda x: len(x.get("href").split("/")) > 1
            and x.get("href").split("/")[1] == "music",
            links,
        )
    )
    links_to_skip = 2
    for i, link in tqdm(enumerate(links), desc="links", position=0, leave=False):
        if i < links_to_skip:
            continue
        music_page = requests.get(root + link.get("href"))
        music_soup = bs4.BeautifulSoup(music_page.content, "html.parser")
        music_links = list(
            filter(
                lambda x: x.get("href").endswith(".mid"),
                music_soup.find_all("a"),
            )
        )
        for music_link in tqdm(
            music_links, desc="music_links", position=1, leave=False
        ):
            music = requests.get(root + link.get("href") + music_link.get("href"))
            file_name = music_link.get("href").split("/")[-1]
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(music.content)
                temp.flush()
                upload_blob(bucket_name, temp.name, f"vg_music/{file_name}")
            time.sleep(0.1)  # avoid rate limiting


if __name__ == "__main__":
    main()
