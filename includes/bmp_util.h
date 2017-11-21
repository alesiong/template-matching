#ifndef BMP_UTIL_H
#define BMP_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uchar;

// Function ReadBMP(): to read a GRAYSCALE image (in 24-bit BMP format)
// bmpName: input, the image file name
// width, height: output, storing the image width and height
// Output float*: a 1-D array that stores the image (row-major)
// Returns 0 if the function fails.

// Example usage: float *image = ReadBMP("test.bmp", &width, &height)
// Note: Put the images at the same folder with your code files;
// or you have to attach the full location of it, such as
//"c:/users/lzh/documents/cuda/cuda/test.bmp" (using slash"/" instead of
// backslash"\")  and remember to free(image) at the end
float *ReadBMP(const char *bmpName, int *width, int *height);

// Function MarkAndSave():
// Read an image from bmpName
// Draw a red rectangle counter with vertices at (X1,Y1) (X2,Y1) (X1,Y2) (X2,Y2)
// on the image  And save the result at outputBmpName (in 24-bit BMP format)
// Example usage: MarkAndSave("I.bmp", 10, 10, 50, 50, "output.bmp");
void MarkAndSave(const char *bmpName, int X1, int Y1, int X2, int Y2,
                 const char *outputBmpName);

#ifdef __cplusplus
}
#endif

#endif
