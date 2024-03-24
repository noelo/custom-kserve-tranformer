from kserve import Model, ModelServer, model_server
from kserve.model import PredictorProtocol
from typing import Dict
import argparse
from tritonclient.grpc.service_pb2 import ModelInferRequest, ModelInferResponse
from tritonclient.grpc import InferResult, InferInput


class KVTransformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.ready = True

    def preprocess(self, request: Dict) -> ModelInferRequest:
        
        inputs = []
        for key,data in request:
            inputs.insert(data)          
        # Transform to KServe v1/v2 inference protocol
        inputs = [{"data": input_tensor} for input_tensor in inputs]
        request = {"instances": inputs}
        return request

    def postprocess(self, infer_response: ModelInferResponse) -> Dict:
        if self.protocol == PredictorProtocol.GRPC_V2.value:
            response = InferResult(infer_response)
            return {"predictions": response.as_numpy("OUTPUT__0").tolist()}
        else:
            return infer_response


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)
parser.add_argument(
    "--protocol", help="The protocol for the predictor", default="v1"
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = KVTransformer(args.model_name, predictor_host=args.predictor_host,
                             protocol=args.protocol)
    ModelServer(workers=1).start([model])