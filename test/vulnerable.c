#include <stdio.h>
#include <string.h>

void function(char *input) {
    char buffer[16];
    
    // Vulnerability: strcpy does not check bounds, potentially allowing an input larger than 16 bytes to overflow the buffer.
    strcpy(buffer, input);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        function(argv[1]);
    } else {
        printf("Please provide an argument.\n");
    }
    return 0;
}