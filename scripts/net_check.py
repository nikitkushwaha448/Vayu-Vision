import requests

def main():
    try:
        r = requests.get('https://api.openaq.org/v2/measurements?limit=1', timeout=10)
        print('status', r.status_code)
        print(r.text[:400])
    except Exception as e:
        print('error', repr(e))

if __name__ == '__main__':
    main()
