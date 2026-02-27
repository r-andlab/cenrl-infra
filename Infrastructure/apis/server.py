from fastapi import FastAPI, Request
import uvicorn
import threading
from Infrastructure.utils.store import MeasurementStore
from Infrastructure.utils.structures import TestPayload, EvalPayload

class MeasurementReceiver:
    def __init__(self, store: MeasurementStore, host="0.0.0.0", port=9000):
        self.store = store
        self.host = host
        self.port = port
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/measurement-done")
        async def measurement_done(req: TestPayload):
            request_id = req.location.country_name  # must match what Go sends
            self.store.record_result(request_id, req)
            return {"status": "ok"}

        @self.app.post("/vp-evaluated")
        async def vp_evaluated(req: EvalPayload):
            print(f"Evaluated vantage point {req.vp}, status is {'OK' if req.template else 'DOWN'}")
            return {"status": "ok"}

        # @self.app.middleware("http")
        # async def log_every_request(request, call_next):
        #     print(">>> INCOMING REQUEST:", request.method, request.url.path)
        #     response = await call_next(request)
        #     print("<<< RESPONSE STATUS:", response.status_code)
        #     return response

    def start_in_background(self):
        def run():
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def debug_start(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="debug")
