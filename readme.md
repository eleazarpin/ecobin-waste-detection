# Deteccion Objetos (Red Neuronal Ecobin)

**Utilizar Python Python 3.7.6**

## Ejecucion en Windows

Descargar Python de https://www.python.org/ftp/python/3.7.6/python-3.7.6.exe

Luego agregar a la variable de entorno Path la siguiente ruta:
```
C:\Users\eleaz\AppData\Local\Programs\Python\Python37\Scripts
```

Crear la carpeta C:\tensorflow y dentro del directorio hacer git clone --branch r1.13.0 https://github.com/tensorflow/models/

Luego crear una nueva variable de entorno llamada PYTHONPATH y agregar las siguientes rutas:
```
C:\tensorflow\models;C:\tensorflow\models\research;C:\tensorflow\models\research\slim
```

Descargar https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip y descomprimir el directorio en el disco C:\

Luego agregar a la variable de entorno Path la siguiente ruta:
```
C:\protoc-3.4.0-win32\bin
```

Abrir el CMD y ejecutar
```
cd C:\tensorflow\models\research
protoc object_detection/protos/*.proto --python_out=.
```

Luego en una terminal powershell dentro del root deteccion-residuos ejecutar:
```powershell
pip install tensorflow==1.13.1
pip install flask
pip install opencv-python
py app.py
```

Una vez corriendo el servicio probar la siguiente request a través de postman:

![ScreenShot](https://github.com/EcoBinUnlam/deteccion-residuos/blob/master/request.png)



## Demonizar servicio en el nuestro servidor linux en EC2

Listar servicios
```bash
pm2 list
```

Demonizar servicio python
```bash
pm2 start app.py --name deteccion-residuos --interpreter python
```

Restartear servicio python
```bash
pm2 restart deteccion-residuos
```

Restartear servicio python
```bash
pm2 delete deteccion-residuos
```
