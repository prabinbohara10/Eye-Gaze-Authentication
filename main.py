import uvicorn
from src.inference_api import app


if __name__ == "__main__":
    print("starting api")
    #killport.kill_ports(ports=[8000])
    uvicorn.run("main:app", host = "0.0.0.0", port= 8001, log_level = "info", reload = True)