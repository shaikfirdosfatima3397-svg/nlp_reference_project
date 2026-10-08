import json
from app.pipeline import SupportPipeline
from app.evaluation import evaluate

if __name__ == "__main__":
    print(json.dumps(evaluate(SupportPipeline()), indent=2))
