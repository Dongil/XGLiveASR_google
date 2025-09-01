import yaml

TRANSLATION_YAML_PATH = "configs/translation.yaml"  # 또는 절대 경로로 지정

def load_translation(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_translation(data, lang='ko', key='Language'):
    if lang not in data:
        print(f"❌ '{lang}' 섹션이 없습니다.")
        return
    section = data[lang]
    if key not in section:
        print(f"❌ '{key}' 키가 '{lang}' 섹션에 없습니다.")
    else:
        print(f"✅ [{lang}] '{key}':", section[key])

def main():
    print("📁 파일 경로:", TRANSLATION_YAML_PATH)

    try:
        data = load_translation(TRANSLATION_YAML_PATH)
    except Exception as e:
        print("❌ YAML 파싱 실패:", e)
        return

    print("\n✅ YAML 파싱 성공")
    print("📌 키 목록 (en):", list(data.get("en", {}).keys())[:5])
    print("📌 키 목록 (ko):", list(data.get("ko", {}).keys())[:5])

    print("\n🔍 번역 테스트:")
    check_translation(data, lang="en", key="Language")
    check_translation(data, lang="ko", key="Language")

if __name__ == "__main__":
    main()
