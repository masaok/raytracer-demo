#include "scene.h"
#include "objects.h"
#include "vect.h"
#include "color.h"

#include <iostream>
#include <fstream>

using namespace std;


int main() {
    
    int Height = 500;
    int Width = 500;
    
    Color_t pix_col(0, 0, 0);
    
    Scene scene = Scene();
    
    Sphere sphere1 (Vec3(200, 250, 220), 50, Color_t(255,100,40),SPECULAR);
	Sphere sphere2 (Vec3(300, 250, 230), 100, Color_t(40, 255, 100), SPECULAR);
	Sphere sphere3 (Vec3(100, 140, 265), 75, Color_t(100, 100, 100), SPECULAR);
	Sphere sphere4(Vec3(300, 2000, 265), 1605, Color_t(200, 200, 200), SPECULAR);
	//Sphere sphere5(Vec3(300, -2000, 265), 1605, Color_t(200, 200, 200), REFLECTIVE);
	Cylinder cylinder1(Vec3(300, 90, 230), Vec3(0,1,0), 50, 275, Color_t(140, 255, 200), SPECULAR);
	Cylinder cylinder2(Vec3(80, 365, 130), Vec3(0.3, -0.4, 1), 50, 275, Color_t(140, 255, 200), SPECULAR);

	Cone cone1(Vec3(300, 90, 230), Vec3(0, 1, 0), 0.95, 275, Color_t(140, 255, 200), SPECULAR);

	Circle circle1(Vec3(300, 90, 230), Vec3(0, 0, 1), 275, Color_t(140, 255, 200), SPECULAR);

	Lightsource light1 = Lightsource(Vec3(0, 0, 0), Color_t(255,255,255));
    
    scene.add(&sphere1);
	/*scene.add(&sphere2);
	scene.add(&sphere3);
	scene.add(&sphere4);
	
	scene.add(&cylinder1);
	scene.add(&cylinder2);

	scene.add(&cone1);
	*/
	scene.add(&circle1);
	
	scene.add(light1);

    ofstream my_Image ("image.ppm");

    if (my_Image.is_open ()) {
        my_Image << "P3\n" << Width << " " << Height << " 255\n";
        for (int i = 0; i < Height; i++) {
            for (int j = 0; j < Width; j++)  {

            pix_col = scene.trace(i, j);


            my_Image << (int)pix_col.r << ' ' << (int)pix_col.g << ' ' << (int)pix_col.b << "\n";
            }
        }
        my_Image.close();
    }
    else
      cout << "Could not open the file";
    
    return 0;
}