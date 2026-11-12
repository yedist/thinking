import src.thinking as think


PATH = r"C:\Users\yedist\PycharmProjects\thinking\temp_models\3Qwen0.6"
REPO_ID = "Qwen/Qwen3-0.6B"


if __name__ == '__main__':
    assert PATH == think.download_model(PATH, REPO_ID)
