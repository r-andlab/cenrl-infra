from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import multiprocessing
import uvicorn
from Infrastructure.utils.structures import TestPayload, EvalPayload

logger = logging.getLogger("uvicorn.error")


def _run_server(measurement_queue, eval_queue, host, port, log_level):
    """Entry point for the server subprocess.

    Runs a fresh FastAPI/Uvicorn instance that puts received payloads onto
    multiprocessing queues so the main process can consume them.
    """
    app = FastAPI()

    # @app.exception_handler(RequestValidationError)
    # async def validation_exception_handler(request: Request, exc: RequestValidationError):
    #     # Log the detailed error message and the request body
    #     logger.error(f"Validation error: {exc.errors()} | Body: {exc.body}")
    #     return JSONResponse(
    #         status_code=422,
    #         content={"detail": exc.errors(), "body": exc.body},
    #     )
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.exception(f"Unhandled error for {request.method} {request.url}:\n{request.body}\n\n")
            raise e

    @app.post("/measurement-done")
    def measurement_done(req: TestPayload):
        measurement_queue.put(req.model_dump())
        return {"status": "ok"}

    @app.post("/vp-evaluated")
    def vp_evaluated(req: EvalPayload):
        eval_queue.put(req.model_dump())
        return {"status": "ok"}

    uvicorn.run(app, host=host, port=port, log_level=log_level)


class MeasurementReceiver:
    def __init__(self, store, eval_store=None,
                 host="0.0.0.0", port=9000):
        self.store = store
        self.eval_store = eval_store
        self.host = host
        self.port = port
        self._measurement_queue = multiprocessing.Queue()
        self._eval_queue = multiprocessing.Queue()
        self._process = None

    def start_in_background(self):
        """Start the HTTP server in a separate process.

        A daemon process ensures it is cleaned up when the main process
        exits.  Incoming payloads are placed onto multiprocessing queues
        and drained into the local stores by :meth:`drain_queues`.
        """
        self._process = multiprocessing.Process(
            target=_run_server,
            args=(
                self._measurement_queue,
                self._eval_queue,
                self.host,
                self.port,
                "warning",
            ),
            daemon=True,
        )
        self._process.start()

    def drain_queues(self):
        """Move all queued payloads from the server process into the
        in-process stores.  Call this at the top of every orchestrator
        tick.
        """
        while not self._measurement_queue.empty():
            data = self._measurement_queue.get_nowait()
            payload = TestPayload(**data)
            country = payload.tag if payload.tag else payload.location.country_name
            self.store.record_result(country, payload)

        while not self._eval_queue.empty():
            data = self._eval_queue.get_nowait()
            payload = EvalPayload(**data)
            if self.eval_store:
                self.eval_store.record(payload)

    def debug_start(self):
        uvicorn.run(
            self._make_app(), host=self.host, port=self.port, log_level="debug"
        )

    def _make_app(self):
        """Build a FastAPI app that writes directly to the stores
        (used only for single-process debug mode).
        """
        app = FastAPI()

        @app.post("/measurement-done")
        def measurement_done(req: TestPayload):
            request_id = req.location.country_name
            self.store.record_result(request_id, req)
            return {"status": "ok"}

        @app.post("/vp-evaluated")
        def vp_evaluated(req: EvalPayload):
            if self.eval_store:
                self.eval_store.record(req)
            return {"status": "ok"}

        return app
