import pandas as pd

df = pd.read_csv('support_tickets/advanced_output.csv')

print('=== STATUS DISTRIBUTION ===')
print(df['Status'].value_counts())

print('\n=== REQUEST TYPE DISTRIBUTION ===')
print(df['Request Type'].value_counts())

print('\n=== ALL TICKETS ===')
for _, row in df.iterrows():
    status = str(row['Status']).upper()
    subject = str(row['Subject'])
    rtype = str(row['Request Type'])
    justification = str(row['Justification'])[:200]
    print(f"[{status}] {subject} | {rtype}")
    print(f"  -> {justification}")
    print()
