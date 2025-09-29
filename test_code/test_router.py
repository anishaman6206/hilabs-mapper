from src.hilabs_mapper.router import route_system

examples = ["Diagnosis", "Procedure", "Lab", "Medicine", "RandomText"]
for et in examples:
    print(f"{et:12} -> {route_system(et)}")