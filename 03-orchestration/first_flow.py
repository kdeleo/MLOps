from prefect import flow, task
import time


@task
def extract():
    print("📥 Extracting data...")
    time.sleep(1)
    return [1, 2, 3, 4, 5]


@task
def transform(data):
    print("🔄 Transforming data...")
    time.sleep(1)
    return [x * 2 for x in data]


@task
def load(data):
    print("📤 Loading data...")
    time.sleep(1)
    print(f"✅ Final result: {data}")


@flow
def etl_flow():
    data = extract()
    transformed = transform(data)
    load(transformed)


if __name__ == "__main__":
    etl_flow()
