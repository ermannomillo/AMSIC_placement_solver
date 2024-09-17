#include <stddef.h>  // Include this for NULL definition

typedef struct {
    int x, y, w, h; 
    int rect_idx;
    int pvar;
} Rectangle;

int can_place_rectangle(double *a, Rectangle *placed_rectangles, int num_rectangles,
                        int x, int y, int w, int h, int rect_idx, int W_max, int H_max, int N) {
    // Bounds check for placement within W_max and H_max
    if (x < 0 || y < 0 || x + w > W_max || y + h > H_max) {
        return 0;  // False, out of bounds
    }

    // Check if a and placed_rectangles are valid
    if (a == NULL || placed_rectangles == NULL) {
        return 0;  // False, invalid pointer
    }

    for (int i = num_rectangles - 1; i >= 0; i--) {
        Rectangle p = placed_rectangles[i];

        // Check if this rectangle has the same index (overlap check)
        if (p.rect_idx == rect_idx) {
            return 0;  // False, same rectangle
        }

        // Bounds check for rect_idx and p.rect_idx to avoid out-of-bounds access in a
        if (rect_idx >= N || p.rect_idx >= N || rect_idx * N + p.rect_idx >= N * N) {
            return 0;  // False, out of bounds
        }

        // Safely access the distance matrix
        double min_d = a[rect_idx * N + p.rect_idx];

        // Check if the rectangles overlap (considering the minimum distance)
        if (!(x + w + min_d <= p.x || x >= p.x + p.w + min_d || 
              y + h + min_d <= p.y || y >= p.y + p.h + min_d)) {
            return 0;  // False, overlap
        }
    }

    return 1;  // True, rectangle can be placed
}

