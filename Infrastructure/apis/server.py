from fastapi import FastAPI, Request
import uvicorn
import threading
from Infrastructure.utils.store import MeasurementStore
from Infrastructure.utils.structures import RequestPayload, ResponseData, LocationData

class MeasurementReceiver:
    def __init__(self, store: MeasurementStore, host="0.0.0.0", port=9000):
        self.store = store
        self.host = host
        self.port = port
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/measurement-done")
        async def measurement_done(req: RequestPayload):
            request_id = req.location.country_name  # must match what Go sends
            self.store.record_result(request_id, req)
            return {"status": "ok"}

    def start_in_background(self):
        def run():
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def debug_start(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="debug")
