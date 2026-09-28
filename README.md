# Random Forest por planta: proyección de los meses restantes del año

App de Streamlit que proyecta, con un Random Forest, el spend por cost bucket y el Recovery de 11 plantas para los meses que faltan del año. Clasifica cada predicción como Riesgo, En línea u Oportunidad contra el forecast de la planta o contra PY.

Las plantas son Mount Royal, Beaver Dam, Champaign, Lowville, Fremont, Muscatine, Jacksonville (Heinz), Cedar Rapids, Mason, Escalon y Holland.

El repositorio no contiene datos. El archivo .xlsb del Performance Model se sube en la app en cada sesión. No subas el .xlsb ni exportaciones de él al repositorio.

## Estructura

```
app.py            todo el código: lectura del .xlsb, modelo, validación, gráficas e interfaz
requirements.txt  dependencias que instala Streamlit Community Cloud
README.md         este archivo
```

Todo el código vive en `app.py` para que el repositorio funcione con solo dos archivos. Así no hay carpetas que puedan quedarse fuera al subir desde la web de GitHub.

## Publicar en GitHub desde la web

1. Abre el repositorio y entra a Add file, luego Upload files.
2. Arrastra `app.py`, `requirements.txt` y `README.md`. Si ya existen, GitHub los reemplaza.
3. Escribe un mensaje de commit y confirma con Commit changes.

Con git en tu computadora:

```
git add app.py requirements.txt README.md
git commit -m "App en un solo archivo"
git push
```

## Desplegar en Streamlit Community Cloud

1. Entra a share.streamlit.io con tu cuenta de GitHub.
2. Crea una app nueva y elige el repositorio, la rama `main` y el archivo `app.py`.
3. En la configuración avanzada elige Python 3.11 o 3.12.
4. Despliega. La primera instalación de dependencias toma unos minutos.

Si la app ya existe, al hacer commit en GitHub se vuelve a desplegar sola. Si no se actualiza, usa Reboot app desde el menú Manage app.

Una app desplegada desde un repo público es pública. Como el repo no lleva datos, lo único que alguien vería es la pantalla que pide el archivo. Aun así, los resultados que la app muestra al subir el .xlsb viven en el servidor de Streamlit durante la sesión. Antes de usarla con datos de la compañía, confirma que tu política de datos lo permite.

## Ejecutar en tu computadora

```
pip install -r requirements.txt
streamlit run app.py
```

## Uso mes a mes

Sube el .xlsb del mes (9+3, 10+2). La app detecta el último mes real con las marcas YTD/YTG de la hoja Actuals y ajusta sola el número de modelos (uno por mes restante). El selector "Último mes real" permite mover el corte hacia atrás, hasta mayo, para revisar cómo habría proyectado el modelo. No permite moverlo hacia adelante, porque esos meses son forecast.

## Metodología (configuración aprobada)

| Componente | Configuración |
|---|---|
| Objetivo | Híbrido: regresión en USD 000's, y la clase se obtiene comparando la predicción contra una referencia |
| Series | 10 cost buckets (Labor, People Compensation, Maintenance, Other Fixed, Utilities, Other VIC, MUV, Other NICC, Faulty, Supply Chain Losses) más Recovery, por planta. Spend neto = suma de las 11 |
| Fuente | Hoja Actuals (año en curso: real hasta el corte, forecast después), hoja PY (año anterior completo), PM Database (Inflation y Gross Savings) |
| Valor a predecir | Índice = valor del mes / promedio de la serie en los 3 meses reales hasta el corte, con piso de USD 10K; se convierte de vuelta a USD 000's |
| Método | Directo: un Random Forest por mes hacia adelante, entrenado con las 121 series (11 plantas × 11 series) juntas |
| Variables | Base: planta, serie, mes, volumen FG del mes, 3 meses previos, tamaño de la serie. Anual: valor y volumen PY del mismo mes (vacíos en filas del año anterior; scikit-learn ≥ 1.4 acepta celdas vacías). Plan: Inflation y Gross Savings. Cada grupo se prende o apaga en la barra lateral |
| Excluidas | Other Inefficiencies (se calcula con los datos reales y le daría la respuesta al modelo) y la hoja Productivity (posible doble conteo con Gross Savings) |
| Volumen futuro | Forecast de la planta, PY del mismo mes, o PY × variación YTD de la planta, con ajuste de ±30% por planta. Hay un aviso cuando el volumen queda fuera del rango histórico, porque el Random Forest no extrapola |
| Clasificación | Riesgo si la predicción supera a la referencia por más del umbral % y del monto mínimo; Oportunidad en el sentido contrario; lo demás, En línea. Convención (Income)/Expense: más alto es más costo |
| Parámetros | Árboles (100 a 1,000) y profundidad máxima en la barra lateral. Fijos: 3 filas mínimas por hoja, fracción de variables 0.5, semilla 42 |
| Validación | Cortes sucesivos (último mes real de enero al mes anterior al corte), WAPE y MAE contra dos métodos simples: promedio de 3 meses, y PY × variación YTD de la serie (razón acotada entre 0.25 y 4). La clasificación se evalúa contra PY |
| Banda | Percentiles 10 y 90 de las predicciones de los árboles. Solo se muestra para una planta y una serie, y no es un intervalo de confianza calibrado |

## Resultado de referencia con el archivo 8+4 (corte ago-26, configuración inicial)

Calculado con scikit-learn 1.9.1, la versión que instala hoy `requirements.txt`. Con otra versión las cifras pueden moverse en décimas.

| Meses hacia adelante | WAPE RF | WAPE promedio 3 meses | WAPE PY × YTD |
|---|---|---|---|
| 1 | 16.1% | 21.5% | 20.5% |
| 2 | 20.5% | 23.4% | 21.7% |
| 3 | 22.4% | 24.4% | 22.2% |
| 4 | 23.1% | 25.0% | 23.7% |
| Total (2,662 predicciones) | 20.0% | 23.3% | 21.8% |

El Random Forest le gana a los dos métodos simples en 6 de 11 plantas, en 4 de 11 series y en 43 de 121 combinaciones planta × serie. En Escalon pierde en las 11 series: WAPE de 69.2% contra 40.7% del método PY × YTD. La clase coincide con la real contra PY en 57.6% de los casos.

Tiempo medido en un contenedor de 2 núcleos: 1 segundo para leer el archivo y entre 71 y 81 segundos para la primera corrida completa (entrenamiento, validación e importancia). Los cambios de volumen, referencia o umbrales tardan de 4 a 6 segundos, porque usan los modelos en caché. En Streamlit Community Cloud los tiempos dependen del hardware asignado.

## Limitaciones y puntos por validar

- Versión del archivo: PM Database dice 8+4, mientras que la celda A1 de Actuals, Other y Productivity dicen 5+7. La app usa las marcas YTD/YTG.
- Escalon: el modelo no capta su estacionalidad con solo 20 meses de historia (su Recovery fue de −5,739 en ago-26 y el forecast lo lleva a −2 en dic-26). Para sep a dic proyecta USD 10.4M más de absorción que el forecast de la planta. Revisa la tabla de validación antes de usar sus clases.
- La validación usa el volumen real de cada mes, así que no incluye el error de pronóstico de volumen.
- Posible traslape entre Gross Savings de PM Database y la hoja Productivity.
- El archivo no indica la unidad del volumen FG.
