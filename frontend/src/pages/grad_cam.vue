<template>
    <div class="page" v-loading="loading" element-loading-background="rgba(0, 0, 0, 0.35)">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="getLayers">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="images[image]" :value='image'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">layer</div>
                <el-select class="value" size="small" v-model="params.layer" @change="update">
                    <el-option v-for="layer in layers" :value="layer.index" :key="layer.index">{{layer.index}} - {{layer.name}} / {{layer.layer}}</el-option>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" type='number' size="small" v-model="params.target"  @change="update"/></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="input">
                <div class="image-wrapper">
                    <el-image :src="params.input">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Input</div>
                </div>
            </div>
            <div class="sliency">
                <div class="image-wrapper">
                    <el-image :src="res.grayscale">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Grayscale</div>
                </div>
                <div class="image-wrapper">
                    <el-image :src="res.colorful">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Grad CAM</div>
                </div>
                <div class="image-wrapper">
                    <el-image :src="res.on_image">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Grad CAM * Image</div>
                </div>
            </div>
            <div class="sliency">
                <div class="image-wrapper">
                    <el-image :src="res.guided_saliecy">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Guided Gradient</div>
                </div>
                <div class="image-wrapper">
                    <el-image :src="res.guided_grad_cam">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Guided Grad CAM</div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            images: {},
            params: {
                model: 'vgg19',
                input: '',
                layer: 37,
                target: null
            },
            layers: {},
            res: {},
            loading: false
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        layers(data) {
            this.layers = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
            this.params.target = this.images[this.params.input]
        },
        response_gradcam(data) {
            this.res = data
            this.loading = false
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
            this.getLayers()
        },
        getLayers() {
            this.$socket.emit('get_layers', { model: this.params.model })
        },
        update() {
            this.loading = true
            this.res = {}
            this.$socket.emit("gradcam", this.params);
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;
        flex-flow: row;

        align-items: center;
        justify-items: center;

        .input {
            flex: 0 0 auto;
        }

        .sliency {
            flex: 1 1 auto;
            display: flex;
            flex-flow: column;
            align-items: center;
            justify-items: center;
        }
    }
}
</style>
