python3 -c "from src.common.solver_input import SolverInput; \
obj = SolverInput.__new__(SolverInput); \
try: \
    obj.unauthorized_attribute = 'leak'; \
    print('❌ FAIL: Memory is still dynamic (__dict__ exists)'); \
except AttributeError: \
    print('✅ SUCCESS: Memory is hardened (__slots__ active)')"