Aplicación para medición de frecuencia cardíaca en cladóceros
Para la obtención de los datos de frecuencia cardíaca (BPM), se desarrolló una herramienta de análisis de video artificial que sustituye el conteo manual por un procesamiento digital de señales, basado en tres puntos principales:
1. Selección de regiones de interés (ROI)
El sistema permite seleccionar de forma manual el área pericárdica del organismo. Dentro de esta zona, se capturan las variaciones constantes de luz y movimiento que ocurren durante el ciclo cardíaco (sístole y diástole).
2. Extracción y multi-análisis de señales
En lugar de depender de una sola variable, la aplicación analiza el video bajo tres criterios simultáneos	 para asegurar la precisión:
Contraste de brillo (Valles): Detecta el oscurecimiento del tejido durante la contracción cardíaca.
Dinámica de movimiento: Mide el desplazamiento físico de las paredes del corazón entre fotogramas consecutivos.
Periodicidad: Se realiza una pre-estimación del ritmo dominante para evitar el subconteo de latidos especialmente rápidos o débiles.
3. Refinamiento y validación
Para eliminar el ruido ambiental (movimiento leve del animal, vibración microscópica o cambios en la iluminación), el sistema aplica un filtro de limpieza de “deriva" o desplazamiento. Esto permite aislar exclusivamente el pulso rítmico. Finalmente, se evalúa cuál de las señales capturadas presenta la mayor regularidad biológica y la selecciona para calcular el valor de BPM final.
Resumen para presentación (3 puntos clave):
1. Automatización: Elimina el sesgo y el error del observador humano en conteos rápidos.
2. Sensibilidad: Capaz de detectar micro-movimientos de sístole que a veces no son perceptibles al ojo humano a velocidad normal.
3. Robustez: Al analizar tanto cambios de luz como movimiento físico, el sistema se adapta a diferentes condiciones de iluminación y transparencia del ejemplar.


#====================================================

1.- Origen de video
Las muestras utilizadas para el desarrollo consisten en videos de un minuto pero se permiten videos de mayor duración. Se sirve de la librería OpenCV para python para el procesamiento y el análisis. Está desarrollada para analizar videos capturados en formato avi, mp4, mov, mkv, se obtiene el total de cuadros y la taza de cuadros por segundo.
El usuario debe seleccionar de forma manual el área del corazón para examinar (ROI, por sus siglas Region of Interest), el sujeto no debe moverse demasiado. El área seleccionada será convertido a escala de grises y se calculará la intensidad media.

Se genera una serie temporal:

    	I=[I1,I2,I3,...,IN]

Donde cada Ik es el brillo promedio del ROI en el frame k. Trabajando bajo la hipótesis de que el corazón del cladócero produce cambios periódicos en la región que afectan el valor medio de la intensidad del ROI. La serie I(t) contiene información del latido, aunque mezclada con ruido.

2.- Construcción de la señal cruda
La salida de la etapa anterior es una señal escalar 1D:

    	x[n]=media de intensidad del ROI en el frame n

La señal puede presentar cambios por la iluminación, postura,ruido por la alta frecuencia.

3.- Centrado de señal
La primera transforación consiste en restar la media, expresado matemáticamente:

	xc​[n]=x[n]−xˉ 
	
Con el objetivo de eliminar el nivel promedio de brillo, centrando la señal al rededor de cero evitando que el valor absoluto del brillo afecte la detección.

4.- Eliminación de tendencia lenta (baseline drift)
Se calcula una “media movil larga” para obtener una versión suavizada de la señal que representa la deriva lenta (baseline) y la señal sin esa deriva (detrended).

	b[n]=”media móvil larga" de xc​[n] 
		xd[n]=xc[n]−b[n] 

Si no se elimina esa tendencia, el detector de picos cuenta mal porque la línea base tendría amplios cambios de nivel, causados por las variaciones de iluminación, desplazamiento del animal dentro del ROI, la compresión del video o movimiento no cardiaco.


5.- Suavizado del ruido rápido
Posteriormente se aplica una “media móvil corta” para reducir las fluctuaciones rápidas no fisiológicas, conservando al estructura periódica principal y mejorando la estabilidad del análisis espectral y de picos. Este suavizado es ligero en comparación y no borra los latidos.

	xs  [n]=”media móvil corta“ de xd [n] 
6.- Filtrado pasa banda cardíaco
Se procede a aplicar un filtro Butterworth, para tener una frecuencia lo más plana posible y sin ondulaciones en el rizado para el análisis.

	Xf [n]=BandPass(xs [n],1.2Hz,4.5Hz) 

Se coloca un límite de ritmo cardiaco esperado entre 1.2Hz (72 BPM) y 4.5Hz (270BPM) con el fin de aternuar las variasiones demasiado lentas y muy rápidas, que se considerarían fuera del rango cardíaco Ajustable para incrementar en situaciones de alto estrés.

7.-Estimación de BPM pro frecuencia dominante
Una vez filtrada la señal, transforma del dominio del tiempo al dominio de la frecuencia. Dentro del rango cardíaco se selecciona la frecuencia con mayor potencia.

	fdom = arg max P(f) 	->	BPMfft  = 60 ⋅ fdom 
		   f

8.- Detección temporal de latidos guiada por al frecuencia dominante
Si la frecuencia dominante es fdom entonces el período estimado es:

		  T =      1    
			fdom
Y en frames:

		  T =    fps    
			fdom

A partir de eso se define una distancia mínima entre picos, la lógica es: Si el corazón tiene un periodo aproximado T, no deberían aparecer dos latidos reales mucho más juntos que eso. Se prueban dos hipótesis, que el latído se vea cono un mínimo o que el latido se vea como un máximo,

9.- Evaluación de picos vs valles
Se comparan ambas soluciones y para cada conjunto de eventos se calcula la regularidad de los intervalos, cercanía al BPM y la prominencia de los eventos. Si los picos detectados están en posiciones p1,p2,...,pk , entonces: 			

	  RRi=pi+1−pi 		
		fps



Y con eso se calcula una medida de variabilidad:
		    CV = σ(RR)
			  μ(RR)
Si el patrón es muy irregular, probablemente no se presente bien el latido.
A partir de los intervalos, se puede estimar:

		BPMpeaks =          60          
			       mediana(RR)
Luego se compara con BPMfft. Si los eventos temporales detectados corresponden a la misma oscilación dominanten observada en el espectro, ambas estimaciones deberían ser consistentes. Los picos más prominentes suelen ser los más fiables.

10.- Fusión de estimación espectral y temporal
Las estimaciones se combinan, bmp_fft derivado del espectro y bmp_peaks derivado del intervalo entre picos. Si ambas coniciden se promedian, en caso de presentar discrepancia se confía en la frecuencia dominante. 

11.- Normalización para la visualización
Antes de devolver la señal a la interfaz se normaliza entre 0 y 1, no se cambia la frecuencia, solo facilita dibujar la curva y marcar los latidos sobre una escala común.

		xn [n] =     xf [n] − min( xf )    
			    max(xf) − min( xf )

12.- Salida
Se devuelve la estimación final del ritmo cardiaco, la señal procesada y normalizada, indices de eventos detectados y frecuencia de muestreo del video.
Los datos resultantes para su posterior análisis se pueden guardar en un archivo csv y generan una imagen que se guarda en en formato png.
