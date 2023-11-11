#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, primitive_op_store) {
    OperationStore store;
    Dimension s1 = store.get<ShiftOp>(dimH, 1)->getInput();
    Dimension s2 = store.get<ShiftOp>(dimH, 1)->getInput();
    ASSERT_EQ(s1, s2);
    ASSERT_EQ(store.get<ShiftOp>(dimH, 1), store.get<ShiftOp>(dimH, 1));
    Dimension
        sL = store.get<ShareOp>(dimH, 1)->getInputL(),
        sR = store.get<ShareOp>(dimH, 1)->getInputR();
    ASSERT_NE(sL, sR);
    ASSERT_NE(s1, sL);
    ASSERT_NE(s1, sR);
}

} // namespace kas
