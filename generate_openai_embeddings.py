import pandas as pd
import multiprocessing as mp
import tqdm
from openai import OpenAI
import numpy as np
import os
import argparse
import multiprocessing


def get_embedding(
    client, cleaned_input, embedding_dimension, model="text-embedding-3-large"
):
    embeddings = []
    openai_reply = client.embeddings.create(
        input=cleaned_input,
        model=model,
        encoding_format="float",
        dimensions=embedding_dimension,
    )
    for row in openai_reply.data:
        assert len(row.embedding) == embedding_dimension
        embeddings.append(row.embedding)

    assert len(cleaned_input) == len(embeddings)
    return embeddings


def get_cleaned_input(df):
    titles = df["title"]
    texts = df["text"]
    cleaned_input = []
    cleaned_titles = []
    cleaned_texts = []
    for title in titles:
        cleaned_titles.append(f"{title}".replace("\n", " "))
    for text in texts:
        cleaned_texts.append(f"{text}".replace("\n", " "))
    for pos, cleaned_title in enumerate(cleaned_titles):
        cleaned_text = cleaned_texts[pos]
        cleaned_input.append(f"{cleaned_title} {cleaned_text}")

    assert len(cleaned_input) == len(titles)
    assert len(cleaned_input) == len(texts)
    return cleaned_input


def process_frame(cleaned_input, row_start, embedding_model, embedding_dimension):
    total_rows = len(cleaned_input)
    embeddings_filename = f"output/embedded_dbpedia_1M_{row_start}_{total_rows}.npy"
    if os.path.exists(embeddings_filename):
        print(f"embeddings file already existed: {embeddings_filename}. skipping...")
    else:
        client = OpenAI()
        # process data frame
        embedings = get_embedding(
            client, cleaned_input, embedding_dimension, model=embedding_model
        )
        with open(embeddings_filename, "wb") as f:
            np.save(f, np.array(embedings))
        assert len(cleaned_input) == len(embedings)
    return total_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="text-embedding-3-large embeddings generator",
    )
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--nrows", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=100)
    parser.add_argument("--nprocesses", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding_dimension", type=int, default=3072)
    parser.add_argument("--csv_filename", type=str, default="input/dbpedia_1M.csv")

    args = parser.parse_args()
    skiprows = args.skiprows
    nrows = args.nrows
    chunksize = args.chunksize
    nprocesses = args.nprocesses
    embedding_model = args.embedding_model
    embedding_dimension = args.embedding_dimension
    csv_filename = args.csv_filename

    if not os.path.exists("output"):
        os.makedirs("output")

    print(f"Splitting work among {nprocesses}...")
    print(
        f"Will read {nrows} rows from {csv_filename} starting a position {skiprows}..."
    )
    reader = pd.read_csv(csv_filename, chunksize=chunksize, nrows=nrows + skiprows)
    pool = mp.Pool(nprocesses)

    funclist = []
    at_row = 0
    print(f"creating chunks of work. each chunk has {chunksize} rows...")
    bar = tqdm.tqdm(total=nrows)
    for df in reader:
        if at_row < skiprows:
            at_row = at_row + len(df)
            continue
        else:
            cleaned_input = get_cleaned_input(df)
            # process each data frame
            f = pool.apply_async(
                process_frame,
                [cleaned_input, at_row, embedding_model, embedding_dimension],
            )
            funclist.append(f)
            bar.update(len(cleaned_input))
            at_row = at_row + len(df)
    bar.close()
    print("\nwaiting for work to be completed...\n")
    result = 0
    bar = tqdm.tqdm(total=nrows)
    for f in funclist:
        processed_rows = f.get()
        result += processed_rows
        bar.update(processed_rows)
    pool.close()
    pool.join()
    print(f"There are {result} rows of data. Final df position was {at_row}...")
