import uvicorn

#Updated new API route
uvicorn.run('api:app',host='127.0.0.1',port=8080,reload=False)
