import pandas as pd
import numpy as np

def clean_dataset(input_file, output_file):
    # 1. Διάβασμα αρχείου με παράβλεψη κατεστραμμένων γραμμών (π.χ. λάθος στήλες)
    # on_bad_lines='skip': αγνοεί γραμμές με περισσότερες/λιγότερες στήλες
    try:
        df = pd.read_csv(input_file, header=None, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Critical error reading {input_file}: {e}")
        return

    original_count = len(df)
    
    # 2. Μετονομασία σε x, y για ευκολία (υποθέτουμε 2 διαστάσεις για clustering)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2] # Κρατάμε μόνο τις 2 πρώτες στήλες
        df.columns = ['x', 'y']
    else:
        print(f"Error: File {input_file} has less than 2 columns.")
        return

    # 3. Μετατροπή σε αριθμούς (coercing errors to NaN)
    # Ότι δεν είναι αριθμός γίνεται NaN (Not a Number)
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # 4. Αφαίρεση γραμμών που έχουν NaN (ήταν κείμενο ή κενά)
    df_clean = df.dropna()
    nan_dropped = original_count - len(df_clean)

    # 5. Αφαίρεση διπλότυπων (Duplicates)
    rows_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_dropped = rows_before_dedup - len(df_clean)

    # 6. Αποθήκευση καθαρού αρχείου
    df_clean.to_csv(output_file, index=False, header=False)

    # Εκτύπωση αποτελεσμάτων
    print(f"--- Report for {input_file} ---")
    print(f"Αρχικές εγγραφές: {original_count}")
    print(f"Διεγραμμένες (non-numeric/bad lines): {nan_dropped}")
    print(f"Διεγραμμένα διπλότυπα: {duplicates_dropped}")
    print(f"Τελικές καθαρές εγγραφές: {len(df_clean)}")
    print(f"Αποθηκεύτηκε στο: {output_file}\n")

# Εκτέλεση για τα δύο αρχεία
clean_dataset('raw_data/data202526a_corrupted.txt', 'data/clean_data_a.csv')
clean_dataset('raw_data/data202526b_corrupted.txt', 'data/clean_data_b.csv')