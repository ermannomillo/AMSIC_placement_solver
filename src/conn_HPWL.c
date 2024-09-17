#include <stdio.h>
#include <float.h>

double conn_HPWL(int E_len, int* E_indices[], double* E_costs, int placed_len, double placed[][6]) {
    // placed is a matrix with 6 columns (x, y, w, h, idx, var)
    
    double total_distance = 0.0;
    double total_weight = 0.0;
    
    if (placed_len > 1) {
        for (int i = 0; i < E_len; i++) {
            int* conn_indices = E_indices[i];  // List of indices
            double cost = E_costs[i];          // Corresponding cost

            // Initialize bounding box variables
            double max_x = -DBL_MAX;
            double max_y = -DBL_MAX;
            double min_x = DBL_MAX;
            double min_y = DBL_MAX;
            
            int valid_connection = 0;  // Track if at least one valid connection exists
           
            // Compute the bounding box for the current connection
            for (int j = 0; conn_indices[j] != -1; j++) {  // -1 marks end of the connection list
                int idx = conn_indices[j];
                
                if (idx < placed_len) {
                    double x = placed[idx][0];
                    double y = placed[idx][1];
                    double w = placed[idx][2];
                    double h = placed[idx][3];
                    
                    // Update min/max values based on the actual bottom-left and top-right corners
                    min_x = (x + w / 2 < min_x) ? (x + w / 2) : min_x;
                    min_y = (y + h / 2 < min_y) ? (y + h / 2) : min_y;
                    max_x = (x + w / 2 > max_x) ? (x + w / 2) : max_x;
                    max_y = (y + h / 2 > max_y) ? (y + h / 2) : max_y;

                    valid_connection = 1;  // Mark as a valid connection
                }
            }

            if (valid_connection) {
                // Calculate the HPWL for this connection
                double distance = (max_x - min_x) + (max_y - min_y);
                total_distance += cost * distance;
                total_weight += cost;
            }
        }
    }
    
    // Return the average HPWL
    return (total_weight != 0) ? (total_distance / total_weight) : 0.0;
}

