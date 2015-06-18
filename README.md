# kinectshop 

folgende Klassen sind erforderlich:

opencv_calib3d2xxd
opencv_core2xxd
opencv_features2d2xxd
opencv_flann2xxd
opencv_highgui2xxd
opencv_imgproc2xxd
opencv_nonfree2xxd
opencv_objdetect2xxd

wichtig ist eine korrekte Linker zuweisung unter Linker -> Allgemein -> zusätzl. Bibliotheksverzeichnisse.

Hierbei muss man auf das jeweilige lib verzeichnis von obencv weisen. 
Bei Visualstudio 2013 in 32bit wäre das z.B. C:/Program Files/opencv/build/x86/vc12/lib
vc12 steht hierbei für den VisualC Compiler von 2012, der in visualstudio 2013 verwendung findet!
Bei Visualstudio 2010 muss vc9 gewählt werden.

Unter debugging muss die Umgebung den Pfad zum entsprechenden .bin file enthalten.
Der Pfad ist analog zur lib zu wählen. Also z.B. C:/Program Files/opencv/build/x86/vc12/bin

Es kann nötig sein unter Linker -> Eingabe -> Zusätzliche Abhängigkeiten die benötigten .lib's explizit zu wählen.
Der Einfachheit habe ich alle Libs dort hineingeschrieben.

Inhalt der Zus. Abh.:  (Hier bei opencv2.49)
opencv_calib3d249d.lib
opencv_contrib249d.lib
opencv_core249d.lib
opencv_features2d249d.lib
opencv_flann249d.lib
opencv_gpu249d.lib
opencv_highgui249d.lib
opencv_imgproc249d.lib
opencv_legacy249d.lib
opencv_ml249d.lib
opencv_nonfree249d.lib
opencv_objdetect249d.lib
opencv_ocl249d.lib
opencv_photo249d.lib
opencv_stitching249d.lib
opencv_superres249d.lib
opencv_ts249d.lib
opencv_video249d.lib
opencv_videostab249d.lib

Achtung: Diese Linkerzuweisungen dürfen nur im Debugmodus gewählt werden.
In Releaseconfig stehen bei opencv andere lib zur Verfügung!

Stand 18.06.2015

-----------------------------------------------------------------------------------------------------
# kinectshop

Aktueller Stand unsere Erkennungssoftware. 
Basis ist opencv 2.4.9, da ich einige Probleme mit 2.4.4 hatte. 
Ihr müsst dazu nur die System Umgebungsvariablen an das neue lib Verzeichnis von opencv 2.4.9 anpassen.

Fertig implementiert:

+ Histogrammvergleich
+ SURF Detector

Probleme:
- Histogrammvergleich ist nicht sonderlich robust (Hintergrund stört)

Es Fehlt:
- Erkennung des Rechtecks für einen zuverlässigen Vergleich der Histogramme
  (Implementierung auf Basis der squares.cpp empfohlen)


Benutze Umgebung:
opencv 2.4.9 / x86 / vc12
 / visual studio ultimate 2013


Stand 26.05.2015 Tobias
