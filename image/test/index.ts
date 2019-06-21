import { assert } from 'chai';

import * as tm from '../src/index';

describe('Beginning', () => {
    it('happens', () => {
        assert.equal(typeof tm, 'object', 'tm should be an object');
        assert.equal(typeof tm.getWebcam, 'function', 'tm.getWebcam should be a function');
        assert.equal(typeof tm.version, 'string', 'tm.version should be a string');
        assert.equal(tm.mobilenet.IMAGE_SIZE, 224, 'IMAGE_SIZE should be 224');
    });
});
