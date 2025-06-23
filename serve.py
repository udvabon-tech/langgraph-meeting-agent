from fastapi import FastAPI
from langserve import add_routes
from graph import build_graph

app = FastAPI()
add_routes(app, build_graph())
