import { assert } from 'chai';

import * as tm from '../src/index';

describe('Module exports', () => {
    it('should contain ', () => {
        assert.typeOf(tm, 'object', 'tm should be an object');
        assert.typeOf(tm.getWebcam, 'function', 'tm.getWebcam should be a function');
        assert.typeOf(tm.version, 'string', 'tm.version should be a string');
        assert.typeOf(tm.CustomMobileNet, 'function');
        assert.typeOf(tm.TeachableMobileNet, 'function');
        assert.typeOf(tm.load, 'function');
        assert.typeOf(tm.loadFromFiles, 'function');
        assert.equal(tm.IMAGE_SIZE, 224, 'IMAGE_SIZE should be 224');
    });
});
