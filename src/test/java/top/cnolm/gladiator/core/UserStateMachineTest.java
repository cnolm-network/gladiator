package top.cnolm.gladiator.core;

import org.junit.Test;
import top.cnolm.gladiator.api.UserState;

import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class UserStateMachineTest {

    @Test
    public void unknownUserDefaultsToAlive() {
        UserStateMachine machine = new UserStateMachine();
        assertEquals(UserState.ALIVE, machine.stateOf(UUID.randomUUID()));
        assertEquals(UserState.ALIVE, machine.stateOf(null));
    }

    @Test
    public void moveToReportsWhetherStateChanged() {
        UserStateMachine machine = new UserStateMachine();
        UUID id = UUID.randomUUID();
        machine.reset(id, UserState.ALIVE);

        assertTrue(machine.moveTo(id, UserState.DEAD));
        assertEquals(UserState.DEAD, machine.stateOf(id));
        assertFalse(machine.moveTo(id, UserState.DEAD));
        assertTrue(machine.moveTo(id, UserState.RESPAWNING));
        assertTrue(machine.moveTo(id, UserState.ALIVE));
    }

    @Test
    public void resetOverridesWithoutTransition() {
        UserStateMachine machine = new UserStateMachine();
        UUID id = UUID.randomUUID();
        machine.reset(id, UserState.ELIMINATED);
        assertEquals(UserState.ELIMINATED, machine.stateOf(id));
        machine.reset(id, UserState.ALIVE);
        assertEquals(UserState.ALIVE, machine.stateOf(id));
    }

    @Test
    public void forgetReleasesState() {
        UserStateMachine machine = new UserStateMachine();
        UUID id = UUID.randomUUID();
        machine.reset(id, UserState.DEAD);
        assertEquals(1, machine.size());
        machine.forget(id);
        assertEquals(0, machine.size());
        assertEquals(UserState.ALIVE, machine.stateOf(id));
    }

    @Test
    public void clearDropsEverything() {
        UserStateMachine machine = new UserStateMachine();
        machine.reset(UUID.randomUUID(), UserState.DEAD);
        machine.reset(UUID.randomUUID(), UserState.ELIMINATED);
        assertEquals(2, machine.size());
        machine.clear();
        assertEquals(0, machine.size());
    }
}
