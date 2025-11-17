import base64, json, numpy as np

def ndarray_to_b64(arr: np.ndarray) -> str:
    payload = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }
    return json.dumps(payload)

def b64_to_ndarray(s: str) -> np.ndarray:
    payload = json.loads(s)
    arr = np.frombuffer(base64.b64decode(payload["data"]), dtype=payload["dtype"])
    return arr.reshape(payload["shape"])
