import pandas as pd
def determine_rain_status(rain):
    """
    Determines the rain status based on the given rainfall amount.

    Args:
        rain (float): The amount of rainfall.

    Returns:
        str: The rain status.
    """

    if rain == 0:
        return 'Không mưa'
    elif 0 < rain <= 0.3:
        return 'Mưa không đáng kể'
    elif 0.3 < rain <= 3:
        return 'Mưa nhỏ'
    elif 3 < rain <= 8:
        return 'Mưa'
    elif 8 < rain <= 25:
        return 'Mưa vừa'
    elif 25 < rain <= 50:
        return 'Mưa to'
    elif rain > 50:
        return 'Mưa rất to'
    else:
        return 'Không xác định'

# Apply the function to the DataFrame
df = pd.read_csv('../data/vd.csv')
test_df = pd.read_csv('../data/new_test_data.csv')
df['rain_status'] = df['rain'].apply(determine_rain_status)
test_df['rain_status'] = test_df['rain'].apply(determine_rain_status)
# In DataFrame sau khi thay thế
print(df)

df.to_csv('../data/vd.csv', encoding='utf-8')

test_df.to_csv('../data/new_test.csv', encoding='utf-8')