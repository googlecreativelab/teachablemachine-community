import { assert } from 'chai';

import * as tm from '../src/index';

describe('Test pose library', () => {
    it('constants are set correctly', () => {
        assert.equal(typeof tm, 'object', 'tm should be an object');
        assert.equal(typeof tm.getWebcam, 'function', 'tm.getWebcam should be a function');
        assert.equal(typeof tm.version, 'string', 'tm.version should be a string');
        assert.equal(tm.version, require('../package.json').version, "version does not match package.json.");
    });
});
