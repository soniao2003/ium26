#!/bin/bash

CUTOFF=${1:-1000}

FILE=$(ls *.csv | head -n 1)

if [ -z "$FILE" ]; then
	echo "Brak pliku CSV w bieżącym katalogu" >&2
	exit 1
fi

OUTPUT="processed_results.txt"

head -n 1 "$FILE" > "$OUTPUT"
tail -n +2 "$FILE" | shuf | head -n "$CUTOFF" | cut -d',' -f1-6 >> "$OUTPUT"

echo "Zakończono. Wynik zapisany w $OUTPUT (CUTOFF=$CUTOFF)"