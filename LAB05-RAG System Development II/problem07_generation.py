from data_loader import load_qa

def find_entry(data):
    return next(d for d in data if d["lang"] == "ทางการ" and d["question"].startswith("การตั้งรหัสผ่านที่ปลอดภัย"))

def bad_generator(context):
    # Simulated failure: the generator unknowingly changes a critical number/condition
    return context.replace("12 ตัวอักษร", "6 ตัวอักษร").replace(
        "โดยไม่ใช้ข้อมูลส่วนตัวที่เดาง่าย", "โดยใช้ข้อมูลส่วนตัวที่จำง่าย"
    )

def grounded_generator(context):
    return context

def run():
    data = load_qa()
    entry = find_entry(data)
    context = entry["answer"]

    print("Question:", entry["question"])
    print("\nRetrieved Context:")
    print(context)

    print("\nBad Generation (weakens the minimum password length and reverses the guidance on personal info):")
    print(bad_generator(context))

    print("\nGrounded Generation (sticks to the original Context):")
    print(grounded_generator(context))

    print("\nCause: Retrieval is correct, but the Generator changes a critical detail (minimum password length)")
    print("and even reverses a safety condition (using vs. not using guessable personal info)")
    print("In a security domain this kind of error can be dangerous; the Prompt must force answers to come from Context only")
    print("and Faithfulness must be evaluated before sending the answer to the user")